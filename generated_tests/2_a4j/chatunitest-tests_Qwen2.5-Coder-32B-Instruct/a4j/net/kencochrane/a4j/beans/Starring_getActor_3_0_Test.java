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

public class Starring_getActor_3_0_Test {

    @InjectMocks
    private Starring starring;

    @Mock
    private ArrayList<String> actors;

    @BeforeEach
    public void setUp() throws Exception {
        MockitoAnnotations.openMocks(this);
        // Initialize the actors field using reflection
        Field actorsField = Starring.class.getDeclaredField("actors");
        actorsField.setAccessible(true);
        actorsField.set(starring, actors);
    }

    @Test
    public void testGetActor_ValidIndex() throws Exception {
        // Arrange
        String[] actorNames = { "Actor1", "Actor2", "Actor3" };
        ArrayList<String> mockActors = new ArrayList<>();
        for (String name : actorNames) {
            mockActors.add(name);
        }
        when(actors.size()).thenReturn(mockActors.size());
        when(actors.get(1)).thenReturn(mockActors.get(1));
        // Act
        String result = starring.getActor(1);
        // Assert
        assertEquals("Actor2", result);
    }

    @Test
    public void testGetActor_IndexTooHigh() throws Exception {
        // Arrange
        when(actors.size()).thenReturn(2);
        // Act
        String result = starring.getActor(5);
        // Assert
        assertNull(result);
    }

    @Test
    public void testGetActor_IndexNegative() throws Exception {
        // Arrange
        when(actors.size()).thenReturn(2);
        // Act
        String result = starring.getActor(-1);
        // Assert
        assertNull(result);
    }

    @Test
    public void testGetActor_EmptyList() throws Exception {
        // Arrange
        when(actors.size()).thenReturn(0);
        // Act
        String result = starring.getActor(0);
        // Assert
        assertNull(result);
    }
}
