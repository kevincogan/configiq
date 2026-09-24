"use client";
import {
  PageSection,
  Title,
  Content,
  Grid,
  GridItem,
  Card,
  CardBody,
  CardTitle,
  Button,
} from "@patternfly/react-core";
import {
  MigrationIcon,
  ChartBarIcon,
  SlidersHIcon,
  CalculatorIcon,
  RouteIcon,
  BoltIcon,
} from "@patternfly/react-icons";
import Link from "next/link";

const tools = [
  {
    title: "Recommend sizing",
    description:
      "Find the optimal GPU configuration for a workload target using the AISimulators engine.",
    href: "/recommend",
    icon: <SlidersHIcon />,
  },
  {
    title: "KV cache calculator",
    description:
      "Calculate KV cache memory requirements for any model on supported GPU systems.",
    href: "/kv-cache",
    icon: <CalculatorIcon />,
  },
  {
    title: "GPU explorer",
    description:
      "Compare GPUs across memory, throughput, and architecture generation.",
    href: "/gpu-explorer",
    icon: <ChartBarIcon />,
  },
  {
    title: "Predict performance",
    description:
      "Estimate TTFT, TPOT, and throughput for your model and parallelism configuration.",
    href: "/predict",
    icon: <BoltIcon />,
  },
  // Hidden pending aicostings API
  // {
  //   title: "Hybrid Savings",
  //   description:
  //     "Model cost savings between cloud, on-premise, and hybrid GPU deployment strategies.",
  //   href: "/hybrid-savings",
  //   icon: <MigrationIcon />,
  // },
  // {
  //   title: "Routing economics",
  //   description:
  //     "Analyze request routing between model tiers to optimize cost vs quality tradeoffs.",
  //   href: "/routing",
  //   icon: <RouteIcon />,
  // },
];

export default function HomePage() {
  return (
    <>
      <PageSection>
        <div>
          <Content>
            <Title headingLevel="h1" size="2xl">
              ConfigIQ
            </Title>
            <Content component="p">
              LLM inference sizing, GPU comparison, and cost modeling for
              engineers and infrastructure teams.
            </Content>
            <Content component="small" style={{ fontStyle: "italic" }}>
              A free community service, provided as-is without warranty. All
              results are best-effort estimates for planning, not guarantees of
              real-world performance.
            </Content>
          </Content>
        </div>
      </PageSection>

      <PageSection>
        <Grid hasGutter md={6} xl={4}>
          {tools.map((tool) => (
            <GridItem key={tool.href}>
              <Card isFullHeight isClickable>
                <CardTitle>
                  <span style={{ marginRight: "0.5rem" }}>{tool.icon}</span>
                  {tool.title}
                </CardTitle>
                <CardBody>
                  <Content>
                    <Content component="p">{tool.description}</Content>
                  </Content>
                  <br />
                  <Button
                    variant="link"
                    isInline
                    component={(props) => <Link href={tool.href} {...props} />}
                  >
                    Open tool →
                  </Button>
                </CardBody>
              </Card>
            </GridItem>
          ))}
        </Grid>
      </PageSection>
    </>
  );
}
