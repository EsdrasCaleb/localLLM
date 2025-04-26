package net.kencochrane.a4j.beans;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;
import java.util.ArrayList;

public class BrowseNode_addSubNode_5_0_Test {

    @Test
    void addSubNode() {
        BrowseNode node = new BrowseNode();
        BrowseNode subNode = new BrowseNode();
        node.addSubNode(subNode);
        assertEquals(1, node.getSubNodes().size());
    }
}
