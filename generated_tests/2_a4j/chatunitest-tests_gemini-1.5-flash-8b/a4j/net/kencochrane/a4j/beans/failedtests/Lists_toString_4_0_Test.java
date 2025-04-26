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

class Lists_toString_4_0_Test {

    private Lists lists;

    @BeforeEach
    void setUp() {
        lists = new Lists();
    }

    @Test
    void testToString_emptyArrayList() {
        assertEquals("lists is null or size 0 \n", lists.toString());
    }

    @Test
    void testToString_nullArrayList() {
        lists = new Lists();
        lists.lists = null;
        assertEquals("lists is null or size 0 \n", lists.toString());
    }

    @Test
    void testToString_nonEmptyArrayList() {
        ArrayList<String> listData = new ArrayList<>(Arrays.asList("list1", "list2", "list3"));
        lists.setListId(listData.toArray(new String[0]));
        String expected = "# of Lists = 3\n" + "list - list1\n" + "list - list2\n" + "list - list3\n";
        assertEquals(expected, lists.toString());
    }

    @Test
    void testToString_withNullElements() {
        ArrayList<String> listData = new ArrayList<>(Arrays.asList("list1", null, "list3"));
        lists.setListId(listData.toArray(new String[0]));
        String expected = "# of Lists = 3\n" + "list - list1\n" + "list - null\n" + "list - list3\n";
        assertEquals(expected, lists.toString());
    }
}
