import UIKit

class MenuViewController: UITableViewController {
  var modes: [Mode] = []
  var selectedMode: Mode = .maskColors

  override var preferredStatusBarStyle: UIStatusBarStyle {
    return .lightContent
  }

  override func tableView(_ tableView: UITableView, numberOfRowsInSection section: Int) -> Int {
    return modes.count
  }

  override func tableView(_ tableView: UITableView, cellForRowAt indexPath: IndexPath) -> UITableViewCell {
    let cell = tableView.dequeueReusableCell(withIdentifier: "Cell", for: indexPath)
    cell.textLabel?.text = modes[indexPath.row].rawValue
    return cell
  }

  override func prepare(for segue: UIStoryboardSegue, sender: Any?) {
    if let cell = sender as? UITableViewCell,
       let indexPath = tableView.indexPath(for: cell) {
      selectedMode = modes[indexPath.row]
    }
  }
}
