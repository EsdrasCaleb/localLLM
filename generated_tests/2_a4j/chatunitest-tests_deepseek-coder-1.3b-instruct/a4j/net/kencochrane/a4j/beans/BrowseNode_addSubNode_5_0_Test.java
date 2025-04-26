package net.kencochrane.a4j.beans;

import java.lang.reflect.Field;
import java.util.ArrayList;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;

public class BrowseNode_addSubNode_5_0_Test {

    @Test
    public void testAddSubNode() throws Exception {
        // Given
        BrowseNode bNode = new BrowseNode();
        BrowseNode sNode = new BrowseNode();
        // When
        Field field = BrowseNode.class.getDeclaredField("subNodes");
        field.setAccessible(true);
        ArrayList subNodes = (ArrayList) field.get(bNode);
        bNode.addSubNode(sNode);
        // Then
        assertTrue(subNodes.contains(sNode));
    }
}
