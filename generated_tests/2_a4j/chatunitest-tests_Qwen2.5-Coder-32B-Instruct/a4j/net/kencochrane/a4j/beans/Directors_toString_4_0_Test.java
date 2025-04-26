package net.kencochrane.a4j.beans;

import java.lang.reflect.Field;
import java.util.ArrayList;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import java.io.Serializable;

@ExtendWith(MockitoExtension.class)
public class Directors_toString_4_0_Test {

    @InjectMocks
    private Directors directors;

    @Mock
    private ArrayList<String> mockDirectors;

    @BeforeEach
    public void setUp() throws Exception {
        MockitoAnnotations.openMocks(this);
        // Injecting the mock ArrayList into the directors field
        Field field = Directors.class.getDeclaredField("directors");
        field.setAccessible(true);
        field.set(directors, mockDirectors);
    }

    @Test
    public void testToStringWithNullDirectors() {
        // Arrange
        // Set the directors object to null
        directors = null;
        // Act
        String result = directors == null ? "Director is null or size 0\n" : directors.toString();
        // Assert
        assertEquals("Director is null or size 0\n", result);
    }

    @Test
    public void testToStringWithEmptyDirectors() {
        // Arrange
        when(mockDirectors.size()).thenReturn(0);
        // Act
        String result = directors.toString();
        // Assert
        assertEquals("Director is null or size 0\n", result);
    }

    @Test
    public void testToStringWithOneDirector() {
        // Arrange
        when(mockDirectors.size()).thenReturn(1);
        when(mockDirectors.get(0)).thenReturn("Steven Spielberg");
        // Act
        String result = directors.toString();
        // Assert
        assertEquals("# of Directors = 1\nDirector - Steven Spielberg\n", result);
    }

    @Test
    public void testToStringWithMultipleDirectors() {
        // Arrange
        when(mockDirectors.size()).thenReturn(2);
        when(mockDirectors.get(0)).thenReturn("Steven Spielberg");
        when(mockDirectors.get(1)).thenReturn("Christopher Nolan");
        // Act
        String result = directors.toString();
        // Assert
        assertEquals("# of Directors = 2\nDirector - Steven Spielberg\nDirector - Christopher Nolan\n", result);
    }
}
