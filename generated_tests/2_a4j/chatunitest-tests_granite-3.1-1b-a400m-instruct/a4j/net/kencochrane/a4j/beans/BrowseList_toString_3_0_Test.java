package net.kencochrane.a4j.beans;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;
import java.util.ArrayList;

public class BrowseList_toString_3_0_Test {

    @Test
    void testToString() {
        // Create an instance of BrowseList
        BrowseList browseList = new BrowseList();
        // Set some nodes
        BrowseNode node1 = new BrowseNode();
        BrowseNode node2 = new BrowseNode();
        BrowseNode node3 = new BrowseNode();
        browseList.setBrowseNode(new BrowseNode[] { node1, node2, node3 });
        // Call toString method
        String result = browseList.toString();
        // Verify the result
        assertEquals("# of nodes = 3\n", result);
        assertEquals("Name: node1\nID: node1\n", result.substring(0, 16));
        assertEquals("Name: node2\nID: node2\n", result.substring(16, 32));
        assertEquals("Name: node3\nID: node3\n", result.substring(32, 48));
    }
}
