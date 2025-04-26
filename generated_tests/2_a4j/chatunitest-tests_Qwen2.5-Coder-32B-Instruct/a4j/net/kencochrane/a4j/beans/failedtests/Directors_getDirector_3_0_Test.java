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
public class Directors_getDirector_3_0_Test {

    @InjectMocks
    private Directors directors;

    @BeforeEach
    public void setUp() throws Exception {
        MockitoAnnotations.openMocks(this);
        // Initialize the directors ArrayList using reflection
        Field field = Directors.class.getDeclaredField("directors");
        field.setAccessible(true);
        field.set(directors, new ArrayList<>());
    }

    @Test
    public void testGetDirector_IndexWithinBounds() throws Exception {
        // Arrange
        ArrayList<String> mockDirectors = new ArrayList<>();
        mockDirectors.add("Director1");
        mockDirectors.add("Director2");
        mockDirectors.add("Director3");
        Field field = Directors.class.getDeclaredField("directors");
        field.setAccessible(true);
        field.set(directors, mockDirectors);
        // Act
        String result = directors.getDirector(1);
        // Assert
        assertEquals("Director2", result);
    }

    @Test
    public void testGetDirector_IndexOutOfBounds() throws Exception {
        // Arrange
        ArrayList<String> mockDirectors = new ArrayList<>();
        mockDirectors.add("Director1");
        mockDirectors.add("Director2");
        mockDirectors.add("Director3");
        Field field = Directors.class.getDeclaredField("directors");
        field.setAccessible(true);
        field.set(directors, mockDirectors);
        // Act
        String result = directors.getDirector(5);
        // Assert
        assertNull(result);
    }

    @Test
    public void testGetDirector_EmptyList() throws Exception {
        ArrayList<String> mockDirectors = new ArrayList<>();
        Field field = Directors.class.getDeclaredField("directors");
        field.setAccessible(true);
        field.set(directors, mockDirectors);
        // Act
        String result = directors.getDirector(0);
        // Assert
        assertNull(result);
    }

    @Test
    public void testGetDirector_NegativeIndex() throws Exception {
        // Arrange
        ArrayList<String> mockDirectors = new ArrayList<>();
        mockDirectors.add("Director1");
        mockDirectors.add("Director2");
        mockDirectors.add("Director3");
        Field field = Directors.class.getDeclaredField("directors");
        field.setAccessible(true);
        field.set(directors, mockDirectors);
        // Act
        String result = directors.getDirector(-1);
        // Assert
        assertNull(result);
    }
}
