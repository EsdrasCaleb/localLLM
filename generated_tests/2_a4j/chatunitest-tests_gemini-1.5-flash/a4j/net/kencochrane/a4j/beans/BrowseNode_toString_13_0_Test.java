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

public class BrowseNode_toString_13_0_Test {

    @Test
    void testToString() throws NoSuchFieldException, IllegalAccessException {
        BrowseNode node1 = new BrowseNode();
        Field browseIdField = BrowseNode.class.getDeclaredField("browseId");
        browseIdField.setAccessible(true);
        browseIdField.set(node1, "123");
        Field browseNameField = BrowseNode.class.getDeclaredField("browseName");
        browseNameField.setAccessible(true);
        browseNameField.set(node1, "Test Node");
        Field modeField = BrowseNode.class.getDeclaredField("mode");
        modeField.setAccessible(true);
        modeField.set(node1, "browse");
        assertEquals("123 - Test Node -- browse", node1.toString());
        BrowseNode node2 = new BrowseNode();
        browseIdField.set(node2, "456");
        browseNameField.set(node2, "Another Node");
        modeField.set(node2, "search");
        assertEquals("456 - Another Node -- search", node2.toString());
        BrowseNode node3 = new BrowseNode();
        browseIdField.set(node3, "");
        browseNameField.set(node3, "");
        modeField.set(node3, "");
        assertEquals(" -  -- ", node3.toString());
        BrowseNode node4 = new BrowseNode();
        browseIdField.set(node4, null);
        browseNameField.set(node4, null);
        modeField.set(node4, null);
        assertEquals("null - null -- null", node4.toString());
    }
}
