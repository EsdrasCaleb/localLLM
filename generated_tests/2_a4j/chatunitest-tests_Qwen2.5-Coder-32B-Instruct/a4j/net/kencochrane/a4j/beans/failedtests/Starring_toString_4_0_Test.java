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

public class Starring_toString_4_0_Test {

    @Mock
    private ArrayList<String> mockActors;

    @InjectMocks
    private Starring starring;

    @BeforeEach
    public void setUp() {
        MockitoAnnotations.openMocks(this);
    }

    @Test
    public void testToStringWithNullActors() throws NoSuchFieldException, IllegalAccessException {
        Field actorsField = Starring.class.getDeclaredField("actors");
        actorsField.setAccessible(true);
        actorsField.set(starring, null);
        String result = starring.toString();
        assertEquals("Actors is null or size 0\n", result);
    }

    @Test
    public void testToStringWithEmptyActors() {
        when(mockActors.size()).thenReturn(0);
        starring.setActor(new String[0]);
        String result = starring.toString();
        assertEquals("Actors is null or size 0\n", result);
    }

    @Test
    public void testToStringWithSingleActor() {
        when(mockActors.size()).thenReturn(1);
        when(mockActors.get(0)).thenReturn("Actor1");
        starring.setActor(new String[] { "Actor1" });
        String result = starring.toString();
        assertEquals("# of Actors = 1\nActor - Actor1\n", result);
    }

    @Test
    public void testToStringWithMultipleActors() {
        when(mockActors.size()).thenReturn(3);
        when(mockActors.get(0)).thenReturn("Actor1");
        when(mockActors.get(1)).thenReturn("Actor2");
        when(mockActors.get(2)).thenReturn("Actor3");
        starring.setActor(new String[] { "Actor1", "Actor2", "Actor3" });
        String result = starring.toString();
        assertEquals("# of Actors = 3\nActor - Actor1\nActor - Actor2\nActor - Actor3\n", result);
    }

    @Test
    public void testToStringWithNullActorInList() {
        when(mockActors.size()).thenReturn(2);
        when(mockActors.get(0)).thenReturn(null);
        when(mockActors.get(1)).thenReturn("Actor2");
        starring.setActor(new String[] { null, "Actor2" });
        String result = starring.toString();
        assertEquals("# of Actors = 2\nActor - \nActor - Actor2\n", result);
    }
}
