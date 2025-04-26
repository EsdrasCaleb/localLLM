package net.kencochrane.a4j.beans;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.ByteArrayOutputStream;
import java.io.PrintStream;
import java.lang.reflect.Field;
import java.util.ArrayList;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import java.io.Serializable;

@ExtendWith(MockitoExtension.class)
public class BrowseNode_printNode_7_1_Test {

    @Mock
    private ArrayList subNodesMock;

    @Mock
    private BrowseNode sNodeMock;

    @InjectMocks
    private BrowseNode browseNode;

    @Test
    public void testPrintNode() throws Exception {
        // Capture the console output
        ByteArrayOutputStream outContent = new ByteArrayOutputStream();
        System.setOut(new PrintStream(outContent));
        // Set up the mock objects
        when(subNodesMock.size()).thenReturn(2);
        when(subNodesMock.get(0)).thenReturn(sNodeMock);
        when(sNodeMock.getBrowseId()).thenReturn("123");
        when(sNodeMock.getBrowseName()).thenReturn("Node1");
        when(subNodesMock.get(1)).thenReturn(sNodeMock);
        when(sNodeMock.getBrowseId()).thenReturn("456");
        when(sNodeMock.getBrowseName()).thenReturn("Node2");
        // Set the subNodes field of browseNode to the mock subNodes
        Field subNodesField = BrowseNode.class.getDeclaredField("subNodes");
        subNodesField.setAccessible(true);
        subNodesField.set(browseNode, subNodesMock);
        // Set the browseId and browseName fields of browseNode
        browseNode.setBrowseId("789");
        browseNode.setBrowseName("RootNode");
        // Invoke the printNode() method
        browseNode.printNode();
        // Reset the console output
        System.setOut(System.out);
        // Assert the console output
        String expectedOutput = "789 - RootNode\n" + "  -- # of subNodes 2 -- \n" + "\n" + "    123 - Node1\n" + "    456 - Node2\n";
        assertEquals(expectedOutput, outContent.toString());
    }
}
