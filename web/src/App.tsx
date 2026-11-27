import { ExplorePanel, TabularDemo } from "@srikarbolleddu/ml-demo-kit";
import { client, isLive } from "./client";

const CASE_STUDY =
  "https://srikarbolleddu.com/projects/sales-forecast-prediction-system";
const REPO = "https://github.com/srikarbolleddu/ml-sales-forecast";
const DATASET = "https://archive.ics.uci.edu/dataset/502/online+retail+ii";

export function App() {
  return (
    <div className="page">
      <nav className="topbar" aria-label="Site">
        <a href={CASE_STUDY}>← Back to the case study</a>
        <a href={REPO} target="_blank" rel="noopener noreferrer">
          View code <span aria-hidden="true">↗</span>
          <span className="visually-hidden"> (opens in a new tab)</span>
        </a>
      </nav>

      <header className="intro">
        <p className="eyebrow">Live demo · Sales forecasting</p>
        <h1>How many units will this product sell next month?</h1>
        <p>
          A UK online gift retailer sells thousands of products, and demand
          roughly doubles before Christmas. This model forecasts each
          product&apos;s next-month sales from its recent history, price and
          season, and was tested only on months it never saw during training.
          Pick a real product below or change any input.
        </p>
      </header>

      {!isLive && (
        <p className="notice" role="note">
          Showing predictions recorded from the model. The live model API isn't
          connected yet, so edited inputs return the closest recorded example.
        </p>
      )}

      <section aria-labelledby="try" className="section">
        <h2 id="try">Try it</h2>
        <TabularDemo client={client} hideHeader />
      </section>

      <section aria-labelledby="explore" className="section">
        <h2 id="explore">Explore the model</h2>
        <ExplorePanel client={client} />
      </section>

      <footer className="footer">
        <p>
          Project from the UT Austin Post-Graduate Program in AI &amp; Machine
          Learning, rebuilt from scratch on an openly licensed dataset.
        </p>
        <p>
          Data:{" "}
          <a href={DATASET} target="_blank" rel="noopener noreferrer">
            Online Retail II
          </a>{" "}
          (Chen, 2019), UCI Machine Learning Repository, CC BY 4.0. Nothing you
          enter is stored.
        </p>
      </footer>
    </div>
  );
}
