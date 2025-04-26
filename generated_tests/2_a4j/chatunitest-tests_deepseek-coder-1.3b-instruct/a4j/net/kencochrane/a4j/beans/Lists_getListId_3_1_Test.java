package net.kencochrane.a4j.beans;

import java.lang.reflect.Field;
import java.util.ArrayList;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;

public class Lists_getListId_3_1_Test {

    private Lists lists;

    private ArrayList mockedList;

    @BeforeEach
    public void setup() {
        lists = new Lists();
        mockedList = Mockito.mock(ArrayList.class);
        lists.lists = mockedList;
    }

    @Test
    public void testGetListId() throws Exception {
        String testString = "test";
        int index = 0;
        when(mockedList.get(index)).thenReturn(testString);
        Field field = Lists.class.getDeclaredField("lists");
        field.setAccessible(true);
        field.set(lists, mockedList);
        String result = lists.getListId(index);
        assertEquals(testString, result);
    }
}
