package net.kencochrane.a4j.beans;

import java.util.ArrayList;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;

public class BrowseNode_toString_13_0_Test {

    @Test
    public void testToString() {
        BrowseNode node = new BrowseNode();
        node.setBrowseId("123");
        node.setBrowseName("Test Node");
        node.setMode("Test Mode");
        // Test normal case
        String expected = "123 - Test Node -- Test Mode";
        String actual = node.toString();
        assertEquals(expected, actual);
        // Test with sub nodes
        ArrayList<BrowseNode> subNodes = new ArrayList<>();
        BrowseNode subNode = new BrowseNode();
        subNode.setBrowseId("123");
        subNode.setBrowseName("Sub Node");
        subNode.setMode("Sub Mode");
        subNodes.add(subNode);
        node.setSubNodes(subNodes);
        expected = "123 - Test Node -- Test Mode\n  -- # of subNodes 1 -- \n    123 - Sub Node -- Sub Mode";
        actual = node.toString();
        assertEquals(expected, actual);
        // Test with null browseId
        node.setBrowseId(null);
        expected = "null - Test Node -- Test Mode";
        actual = node.toString();
        assertEquals(expected, actual);
        // Test with null browseName
        node.setBrowseName(null);
        expected = "123 - null -- Test Mode";
        actual = node.toString();
        assertEquals(expected, actual);
        // Test with null mode
        node.setMode(null);
        expected = "123 - Test Node -- null";
        actual = node.toString();
        assertEquals(expected, actual);
    }
}
