package net.kencochrane.a4j.beans;

import java.util.ArrayList;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.lang.reflect.Field;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import java.io.Serializable;

@ExtendWith(MockitoExtension.class)
public class Lists_getListId_3_0_Test {

    @InjectMocks
    private Lists lists;

    @Mock
    private ArrayList<String> mockLists;

    @BeforeEach
    public void setUp() throws NoSuchFieldException, IllegalAccessException {
        Field listsField = Lists.class.getDeclaredField("lists");
        listsField.setAccessible(true);
        listsField.set(lists, mockLists);
    }

    @Test
    public void testGetListId_ValidIndex() {
        when(mockLists.size()).thenReturn(3);
        when(mockLists.get(1)).thenReturn("list2");
        String result = lists.getListId(1);
        assertEquals("list2", result);
    }

    @Test
    public void testGetListId_InvalidIndex() {
        when(mockLists.size()).thenReturn(3);
        String result = lists.getListId(3);
        assertNull(result);
    }

    @Test
    public void testGetListId_EmptyList() {
        when(mockLists.size()).thenReturn(0);
        String result = lists.getListId(0);
        assertNull(result);
    }
}
