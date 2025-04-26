package net.kencochrane.a4j.beans;

import java.lang.reflect.Field;
import java.util.ArrayList;
import java.util.Arrays;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;

@ExtendWith(MockitoExtension.class)
public class Starring_getActor_3_0_Test {

    @Mock
    private Starring starring;

    @Captor
    private ArgumentCaptor<Integer> indexCaptor;

    @BeforeEach
    public void setUp() throws Exception {
        ArrayList<String> actorsList = new ArrayList<>(Arrays.asList("Actor1", "Actor2", "Actor3"));
        when(starring.getActorsArray()).thenReturn(actorsList);
    }

    @Test
    public void testGetActorWithinBounds() {
        when(starring.getActor(indexCaptor.capture())).thenReturn("Actor2");
        String actor = starring.getActor(1);
        assertEquals("Actor2", actor);
        assertEquals(1, indexCaptor.getValue());
    }

    @Test
    public void testGetActorOutOfBounds() {
        when(starring.getActor(indexCaptor.capture())).thenReturn(null);
        String actor = starring.getActor(5);
        assertNull(actor);
        assertEquals(2, indexCaptor.getAllValues().size());
        assertEquals(1, indexCaptor.getValue());
    }

    @Test
    public void testGetActorEmptyList() throws Exception {
        when(starring.getActorsArray()).thenReturn(new ArrayList<>());
        when(starring.getActor(indexCaptor.capture())).thenReturn(null);
        String actor = starring.getActor(0);
        assertNull(actor);
        assertEquals(1, indexCaptor.getAllValues().size());
        assertEquals(0, indexCaptor.getValue());
    }
}
