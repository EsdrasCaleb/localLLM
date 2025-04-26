package net.kencochrane.a4j.beans;

import java.io.ByteArrayOutputStream;
import java.io.PrintStream;
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

    @BeforeEach
    public void setUp() {
        MockitoAnnotations.openMocks(this);
        browseNode.setBrowseId("1");
        browseNode.setBrowseName("Root");
        browseNode.setMode("Mode1");
        ArrayList<BrowseNode> subNodes = new ArrayList<>();
        BrowseNode subNode1 = new BrowseNode();
        subNode1.setBrowseId("1.1");
        subNode1.setBrowseName("Sub1");
        BrowseNode subNode2 = new BrowseNode();
        subNode2.setBrowseId("1.2");
        subNode2.setBrowseName("Sub2");
        subNodes.add(subNode1);
        subNodes.add(subNode2);
        browseNode.setSubNodes(subNodes);
    }

    @Test
    public void testPrintNode() {
        ByteArrayOutputStream outContent = new ByteArrayOutputStream();
        System.setOut(new PrintStream(outContent));
        browseNode.printNode();
        String expectedOutput = "1 - Root\n" + "  -- # of subNodes 2 -- \n" + "    1.1 - Sub1\n" + "    1.2 - Sub2\n";
        assertEquals(expectedOutput, outContent.toString());
    }

    @Test
    public void testPrintNodeNoSubNodes() {
        ByteArrayOutputStream outContent = new ByteArrayOutputStream();
        System.setOut(new PrintStream(outContent));
        browseNode.setSubNodes(new ArrayList<>());
        browseNode.printNode();
        String expectedOutput = "1 - Root\n";
        assertEquals(expectedOutput, outContent.toString());
    }
}
