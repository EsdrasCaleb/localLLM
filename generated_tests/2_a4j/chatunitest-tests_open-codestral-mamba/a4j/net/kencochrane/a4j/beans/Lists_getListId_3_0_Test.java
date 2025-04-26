package net.kencochrane.a4j.beans;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.lang.reflect.Field;
import java.util.ArrayList;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import java.io.Serializable;

@ExtendWith(MockitoExtension.class)
public class Lists_getListId_3_0_Test {

    @Mock
    private ArrayList listsMock;

    @InjectMocks
    private Lists lists;

    @BeforeEach
    public void setUp() throws NoSuchFieldException, IllegalAccessException {
        Field listsField = Lists.class.getDeclaredField("lists");
        listsField.setAccessible(true);
        listsField.set(lists, listsMock);
    }

    @Test
    public void testGetListIdWithValidIndex() {
        when(listsMock.get(0)).thenReturn("element1");
        when(listsMock.size()).thenReturn(1);
        String result = lists.getListId(0);
        assertEquals("element1", result);
    }

    @Test
    public void testGetListIdWithInvalidIndex() {
        when(listsMock.size()).thenReturn(1);
        String result = lists.getListId(1);
        assertEquals(null, result);
    }
}
