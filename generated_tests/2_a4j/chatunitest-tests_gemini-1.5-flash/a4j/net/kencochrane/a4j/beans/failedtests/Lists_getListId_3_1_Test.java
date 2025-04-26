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

public class Lists_getListId_3_1_Test {

    private Lists lists;

    @BeforeEach
    void setUp() {
        lists = new Lists();
    }

    @Test
    void testGetListId_validIndex() {
        String[] newListId = { "123", "456", "789" };
        lists.setListId(newListId);
        assertEquals("456", lists.getListId(1));
    }

    @Test
    void testGetListId_indexOutOfBounds() {
        String[] newListId = { "123", "456", "789" };
        lists.setListId(newListId);
        assertNull(lists.getListId(3));
    }

    @Test
    void testGetListId_emptyLists() {
        String[] newListId = {};
        lists.setListId(newListId);
        assertNull(lists.getListId(0));
    }

    @Test
    void testGetListId_negativeIndex() {
        String[] newListId = { "123", "456", "789" };
        lists.setListId(newListId);
        assertNull(lists.getListId(-1));
    }

    @Test
    void testListId_NullInput() {
        lists.setListId(null);
        assertNull(lists.getListId(0));
    }
}
