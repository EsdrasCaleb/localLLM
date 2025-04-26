package net.kencochrane.a4j.beans;

import java.util.ArrayList;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;

public class Directors_toString_4_0_Test {

    private Directors directors;

    @BeforeEach
    public void setUp() {
        directors = new Directors();
    }

    @Test
    public void testToString_WhenDirectorsIsNull() {
        // Arrange
        // directors is not set, so it should be null by default.
        // Act
        String result = directors.toString();
        // Assert
        assertEquals("Director is null or size 0\n", result);
    }

    @Test
    public void testToString_WhenDirectorsIsEmpty() {
        // Arrange
        directors.setDirector(new String[0]);
        // Act
        String result = directors.toString();
        // Assert
        assertEquals("Director is null or size 0\n", result);
    }

    @Test
    public void testToString_WhenDirectorsHasOneDirector() {
        // Arrange
        directors.setDirector(new String[] { "Christopher Nolan" });
        // Act
        String result = directors.toString();
        // Assert
        assertEquals("# of Directors = 1\nDirector - Christopher Nolan\n", result);
    }

    @Test
    public void testToString_WhenDirectorsHasMultipleDirectors() {
        // Arrange
        directors.setDirector(new String[] { "Christopher Nolan", "Steven Spielberg", "Martin Scorsese" });
        // Act
        String result = directors.toString();
        // Assert
        assertEquals("# of Directors = 3\nDirector - Christopher Nolan\nDirector - Steven Spielberg\nDirector - Martin Scorsese\n", result);
    }
}
