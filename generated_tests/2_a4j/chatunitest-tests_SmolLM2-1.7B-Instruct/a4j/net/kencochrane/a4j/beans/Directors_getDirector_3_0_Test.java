package net.kencochrane.a4j.beans;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;
import java.util.ArrayList;

@ExtendWith(MockitoExtension.class)
public class Directors_getDirector_3_0_Test {

    @Mock
    private ArrayList<String> directors;

    @BeforeEach
    public void setup() {
        directors = new ArrayList<>();
    }

    @Test
    public void testGetDirector() {
        // Arrange
        directors.add("Director1");
        directors.add("Director2");
        directors.add("Director3");
        // Act
        String director = directors.get(1);
        // Assert
        assertEquals("Director2", director);
    }

    @Test
    public void testGetDirectorOutOfRange() {
        // Arrange
        directors.add("Director1");
        directors.add("Director2");
        directors.add("Director3");
        // Act and Assert
        assertThrows(IndexOutOfBoundsException.class, () -> directors.get(3));
    }
}
