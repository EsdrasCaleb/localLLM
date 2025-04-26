package net.kencochrane.a4j.beans;

import java.lang.reflect.Field;
import java.util.ArrayList;
import java.util.Arrays;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;

class BrowseList_toString_3_0_Test {

    @Test
    void testToString_emptyList() {
        BrowseList browseList = new BrowseList();
        assertEquals("No nodes\n", browseList.toString());
    }

    @Test
    void testToString_nullList() {
        BrowseList browseList = new BrowseList();
        try {
            Field nodesField = BrowseList.class.getDeclaredField("nodes");
            nodesField.setAccessible(true);
            nodesField.set(browseList, null);
        } catch (NoSuchFieldException | IllegalAccessException e) {
            fail("Failed to set nodes field: " + e.getMessage());
        }
        assertEquals("No nodes\n", browseList.toString());
    }

    // Dummy BrowseNode class for compilation
    static class BrowseNode {

        public String getBrowseName() {
            return "";
        }

        public String getBrowseId() {
            return "";
        }
    }
}
