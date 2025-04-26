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

    private BrowseNode browseNode;

    private final ByteArrayOutputStream outputStreamCaptor = new ByteArrayOutputStream();

    @BeforeEach
    public void setUp() {
        browseNode = new BrowseNode();
        System.setOut(new PrintStream(outputStreamCaptor));
    }

    @Test
    public void testPrintNode_WithNoSubNodes() {
        browseNode.setBrowseId("1");
        browseNode.setBrowseName("Root Node");
        browseNode.printNode();
        String expectedOutput = "1 - Root Node\n";
        assertEquals(expectedOutput, outputStreamCaptor.toString().trim());
    }

    @Test
    public void testPrintNode_WithSubNodes() {
        browseNode.setBrowseId("1");
        browseNode.setBrowseName("Root Node");
        ArrayList<BrowseNode> subNodes = new ArrayList<>();
        BrowseNode subNode1 = Mockito.mock(BrowseNode.class);
        Mockito.when(subNode1.getBrowseId()).thenReturn("1.1");
        Mockito.when(subNode1.getBrowseName()).thenReturn("Child Node 1");
        BrowseNode subNode2 = Mockito.mock(BrowseNode.class);
        Mockito.when(subNode2.getBrowseId()).thenReturn("1.2");
        Mockito.when(subNode2.getBrowseName()).thenReturn("Child Node 2");
        subNodes.add(subNode1);
        subNodes.add(subNode2);
        browseNode.setSubNodes(subNodes);
        browseNode.printNode();
        String expectedOutput = "1 - Root Node\n" + "  -- # of subNodes 2 -- \n" + "    1.1 - Child Node 1\n" + "    1.2 - Child Node 2";
        assertEquals(expectedOutput, outputStreamCaptor.toString().trim());
    }
}
