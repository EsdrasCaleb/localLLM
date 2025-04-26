package net.kencochrane.a4j.beans;

import java.lang.reflect.Method;
import java.util.ArrayList;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;

public class BrowseNode_printNode_7_0_Test {

    @InjectMocks
    private BrowseNode browseNode;

    @Mock
    private BrowseNode mockSubNode;

    @BeforeEach
    public void setUp() {
        MockitoAnnotations.openMocks(this);
        browseNode.setBrowseId("123");
        browseNode.setBrowseName("RootNode");
        browseNode.setMode("NORMAL");
    }

    @Test
    public void testPrintNodeNoSubNodes() throws Exception {
        // Capture the output to verify
        var outContent = new java.io.ByteArrayOutputStream();
        System.setOut(new java.io.PrintStream(outContent));
        // Call the method under test
        browseNode.printNode();
        // Verify the output
        String expectedOutput = "123 - RootNode\n";
        assert outContent.toString().equals(expectedOutput);
    }

    @Test
    public void testPrintNodeWithSubNodes() throws Exception {
        // Prepare sub-nodes
        ArrayList<BrowseNode> subNodes = new ArrayList<>();
        BrowseNode subNode1 = new BrowseNode();
        subNode1.setBrowseId("456");
        subNode1.setBrowseName("SubNode1");
        BrowseNode subNode2 = new BrowseNode();
        subNode2.setBrowseId("789");
        subNode2.setBrowseName("SubNode2");
        subNodes.add(subNode1);
        subNodes.add(subNode2);
        browseNode.setSubNodes(subNodes);
        // Capture the output to verify
        var outContent = new java.io.ByteArrayOutputStream();
        System.setOut(new java.io.PrintStream(outContent));
        // Call the method under test
        browseNode.printNode();
        // Verify the output
        String expectedOutput = "123 - RootNode\n  -- # of subNodes 2 -- \n\n    456 - SubNode1\n    789 - SubNode2\n";
        assert outContent.toString().equals(expectedOutput);
    }
}
