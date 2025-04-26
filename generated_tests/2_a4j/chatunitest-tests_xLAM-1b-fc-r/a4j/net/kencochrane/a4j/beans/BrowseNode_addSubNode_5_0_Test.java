package net.kencochrane.a4j.beans;

import java.util.ArrayList;
import java.util.Arrays;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;

public class BrowseNode_addSubNode_5_0_Test {

    @Test
    public void testAddSubNode() {
        BrowseNode parentNode = new BrowseNode();
        BrowseNode subNode = new BrowseNode();
        parentNode.addSubNode(subNode);
        assertEquals(Arrays.asList(subNode), parentNode.getSubNodes());
    }
}
