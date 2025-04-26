package net.kencochrane.a4j.beans;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;
import java.util.ArrayList;

public class Lists_toString_4_0_Test {

    @Test
    void testToString() {
        Lists lists = new Lists();
        lists.setListId(new String[] { "list1", "list2" });
        String expected = "# of Lists = 2\nlist - list1\nlist - list2\n";
        String actual = lists.toString();
        assertEquals(expected, actual);
    }
}
