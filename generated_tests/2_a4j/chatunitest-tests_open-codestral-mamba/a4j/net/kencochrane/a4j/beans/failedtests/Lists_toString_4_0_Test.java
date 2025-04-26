package net.kencochrane.a4j.beans;

import java.util.ArrayList;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;

public class Lists_toString_4_0_Test {

    @Mock
    private ArrayList<String> lists;

    @InjectMocks
    private Lists list;

    @BeforeEach
    public void setUp() {
        MockitoAnnotations.initMocks(this);
    }

    @Test
    public void testToString() {
        when(lists.size()).thenReturn(3);
        when(lists.get(0)).thenReturn("list1");
        when(lists.get(1)).thenReturn("list2");
        when(lists.get(2)).thenReturn("list3");
        String expected = "# of Lists = 3\n" + "list - list1\n" + "list - list2\n" + "list - list3\n";
        assertEquals(expected, list.toString());
    }

    @Test
    public void testToStringEmptyList() {
        when(lists.size()).thenReturn(0);
        String expected = "lists is null or size 0\n";
        assertEquals(expected, list.toString());
    }

    @Test
    public void testToStringNullList() {
        when(lists.size()).thenReturn(3);
        when(lists.get(0)).thenReturn(null);
        when(lists.get(1)).thenReturn(null);
        when(lists.get(2)).thenReturn(null);
        String expected = "# of Lists = 3\n" + "list - null\n" + "list - null\n" + "list - null\n";
        assertEquals(expected, list.toString());
    }
}
