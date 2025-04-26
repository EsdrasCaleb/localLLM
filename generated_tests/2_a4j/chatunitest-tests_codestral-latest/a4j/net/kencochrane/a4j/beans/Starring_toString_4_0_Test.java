package net.kencochrane.a4j.beans;

import java.util.ArrayList;
import java.util.Arrays;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;

class Starring_toString_4_0_Test {

    @InjectMocks
    private Starring starring;

    @Mock
    private ArrayList<String> actors;

    @BeforeEach
    void setUp() {
        MockitoAnnotations.openMocks(this);
    }

    @Test
    void testToStringWithActors() {
        when(actors.size()).thenReturn(2);
        when(actors.get(0)).thenReturn("Actor1");
        when(actors.get(1)).thenReturn("Actor2");
        String expected = "# of Actors = 2\nActor - Actor1\nActor - Actor2\n";
        String result = starring.toString();
        assertEquals(expected, result);
    }

    @Test
    void testToStringWithEmptyActors() {
        starring.setActor(new String[0]);
        String expected = "Actors is null or size 0\n";
        String result = starring.toString();
        assertEquals(expected, result);
    }
}
