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

public class Lists_toString_4_0_Test {

    @Test
    void testToString_NullList() {
        Lists lists = new Lists();
        assertEquals("lists is null or size 0 \n", lists.toString());
    }

    @Test
    void testToString_EmptyList() {
        Lists lists = new Lists();
        try {
            Field listsField = Lists.class.getDeclaredField("lists");
            listsField.setAccessible(true);
            listsField.set(lists, new ArrayList<>());
        } catch (NoSuchFieldException | IllegalAccessException e) {
            fail("Failed to access lists field: " + e.getMessage());
        }
        assertEquals("# of Lists = 0\n", lists.toString());
    }

    @Test
    void testToString_NonEmptyList() {
        Lists lists = new Lists();
        try {
            Field listsField = Lists.class.getDeclaredField("lists");
            listsField.setAccessible(true);
            listsField.set(lists, new ArrayList<>(Arrays.asList("list1", "list2", "list3")));
        } catch (NoSuchFieldException | IllegalAccessException e) {
            fail("Failed to access lists field: " + e.getMessage());
        }
        String expected = "# of Lists = 3\n" + "list - list1\n" + "list - list2\n" + "list - list3\n";
        assertEquals(expected, lists.toString());
    }

    @Test
    void testToString_ListWithNullElement() {
        Lists lists = new Lists();
        ArrayList<String> list = new ArrayList<>();
        list.add("list1");
        list.add(null);
        list.add("list3");
        try {
            Field listsField = Lists.class.getDeclaredField("lists");
            listsField.setAccessible(true);
            listsField.set(lists, list);
        } catch (NoSuchFieldException | IllegalAccessException e) {
            fail("Failed to access lists field: " + e.getMessage());
        }
        String expected = "# of Lists = 3\n" + "list - list1\n" + "list - null\n" + "list - list3\n";
        assertEquals(expected, lists.toString());
    }

    @Test
    void testToString_ListWithEmptyElement() {
        Lists lists = new Lists();
        ArrayList<String> list = new ArrayList<>();
        list.add("list1");
        list.add("");
        list.add("list3");
        try {
            Field listsField = Lists.class.getDeclaredField("lists");
            listsField.setAccessible(true);
            listsField.set(lists, list);
        } catch (NoSuchFieldException | IllegalAccessException e) {
            fail("Failed to access lists field: " + e.getMessage());
        }
        String expected = "# of Lists = 3\n" + "list - list1\n" + "list - \n" + "list - list3\n";
        assertEquals(expected, lists.toString());
    }
}
