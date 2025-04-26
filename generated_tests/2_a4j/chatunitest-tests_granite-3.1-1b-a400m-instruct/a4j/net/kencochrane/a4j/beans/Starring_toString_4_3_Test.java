package net.kencochrane.a4j.beans;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;
import java.util.ArrayList;

public class Starring_toString_4_3_Test {

    @Test
    void testToString() {
        // Arrange
        Starring starring = new Starring();
        String[] actors = { "Tom Hanks", "Daniel Day-Lewis", "Leonardo DiCaprio" };
        // Act
        String output = starring.toString();
        // Assert
        assertEquals("# of Actors = 3\nActor - Tom Hanks\nActor - Daniel Day-Lewis\nActor - Leonardo DiCaprio", output);
    }

    @Test
    void testToStringNull() {
        // Arrange
        Starring starring = new Starring();
        String[] actors = null;
        // Act
        String output = starring.toString();
        // Assert
        assertEquals("Actors is null or size 0", output);
    }

    @Test
    void testToStringEmptyActors() {
        // Arrange
        Starring starring = new Starring();
        String[] actors = {};
        // Act
        String output = starring.toString();
        // Assert
        assertEquals("Actors is null or size 0", output);
    }
}
