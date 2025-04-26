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

    @InjectMocks
    private Lists lists;

    @BeforeEach
    void setUp() {
        MockitoAnnotations.openMocks(this);
    }

    @Test
    void testToStringWhenListsIsNull() {
        lists.setListId(null);
        String expected = "lists is null or size 0 \n";
        assertEquals(expected, lists.toString());
    }

    @Test
    void testToStringWhenListsIsEmpty() {
        lists.setListId(new String[0]);
        String expected = "lists is null or size 0 \n";
        assertEquals(expected, lists.toString());
    }

    @Test
    void testToStringWhenListsHasElements() {
        String[] listIds = { "list1", "list2", "list3" };
        lists.setListId(listIds);
        String expected = "# of Lists = 3\nlist - list1\nlist - list2\nlist - list3\n";
        assertEquals(expected, lists.toString());
    }
}
