package net.kencochrane.a4j.beans;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;
import java.util.ArrayList;

class BrowseNode_toString_13_2_Test {

    @Test
    public void testToString() {
        BrowseNode node = new BrowseNode();
        node.setBrowseId("123");
        node.setBrowseName("Sample Node");
        node.setMode("Read");
        node.setParentNodes(new ArrayList<>());
        node.addSubNode(new BrowseNode());
        node.addSubNode(new BrowseNode());
        String expectedOutput = "123 - Sample Node -- Read\n  -- # of subNodes 2 --\n";
        assertEquals(expectedOutput, node.toString());
    }
}
