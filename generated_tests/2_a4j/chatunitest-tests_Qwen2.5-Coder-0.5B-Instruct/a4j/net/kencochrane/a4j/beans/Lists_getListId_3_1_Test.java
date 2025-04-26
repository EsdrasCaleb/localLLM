package net.kencochrane.a4j.beans;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;
import java.util.ArrayList;

class Lists_getListId_3_1_Test {

    private Lists lists;

    @BeforeEach
    public void setUp() {
        lists = Mockito.mock(Lists.class);
    }

    @Test
    public void testGetListId() {
        Mockito.when(lists.getListId(0)).thenReturn("id1");
        Mockito.when(lists.getListId(1)).thenReturn("id2");
        String id1 = lists.getListId(0);
        String id2 = lists.getListId(1);
        assertEquals("id1", id1);
        assertEquals("id2", id2);
    }
}
