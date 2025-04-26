package net.kencochrane.a4j.beans;

import java.util.ArrayList;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;

class Starring_getActor_3_0_Test {

    @InjectMocks
    private Starring starring;

    @Mock
    private ArrayList<String> actors;

    @BeforeEach
    void setUp() {
        MockitoAnnotations.openMocks(this);
        starring.actors = actors;
    }

    @Test
    void testGetActorValidIndex() {
        when(actors.size()).thenReturn(3);
        when(actors.get(1)).thenReturn("Actor 2");
        String result = starring.getActor(1);
        assertEquals("Actor 2", result);
    }

    @Test
    void testGetActorInvalidIndex() {
        when(actors.size()).thenReturn(3);
        String result = starring.getActor(3);
        assertNull(result);
    }

    @Test
    void testGetActorEmptyList() {
        when(actors.size()).thenReturn(0);
        String result = starring.getActor(0);
        assertNull(result);
    }
}
