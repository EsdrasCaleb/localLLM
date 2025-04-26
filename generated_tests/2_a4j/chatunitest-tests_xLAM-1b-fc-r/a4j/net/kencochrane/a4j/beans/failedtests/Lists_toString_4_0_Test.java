package net.kencochrane.a4j.beans;

import java.util.ArrayList;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;

class Lists_toString_4_0_Test {

    @Test
    void testToString() {
        Lists lists = new Lists();
        lists.setListId(new String[] { "list1", "list2", "list3" });
        String expectedOutput = "# of Lists = 3\n" + "list - list1\n" + "list - list2\n" + "list - list3\n";
        assertEquals(expectedOutput, lists.toString());
    }

    @Test
    void testToStringWithEmptyList() {
        Lists lists = new Lists();
        String expectedOutput = "# of Lists = 0\n";
        assertEquals(expectedOutput, lists.toString());
    }

    @Test
    void testToStringWithNullList() {
        Lists lists = new Lists();
        lists.setListId(null);
        String expectedOutput = "# of Lists = 0\n";
        assertEquals(expectedOutput, lists.toString());
    }
}
