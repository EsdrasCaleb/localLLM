package net.kencochrane.a4j.beans;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;
import java.util.ArrayList;

class Starring_toString_4_2_Test {

    @Test
    public void testToStringWithNoActors() {
        // Arrange
        Starring starring = new Starring();
        // Act
        String result = starring.toString();
        // Assert
        assertEquals("Actors is null or size 0\n", result);
    }

    @Test
    public void testToStringWithOneActor() {
        // Arrange
        Starring starring = new Starring();
        starring.setActor(new String[] { "John Doe" });
        // Act
        String result = starring.toString();
        // Assert
        assertEquals("# of Actors = 1\nActor - John Doe\n", result);
    }

    @Test
    public void testToStringWithMultipleActors() {
        // Arrange
        Starring starring = new Starring();
        starring.setActor(new String[] { "John Doe", "Jane Smith", "Mike Johnson" });
        // Act
        String result = starring.toString();
        // Assert
        assertEquals("# of Actors = 3\nActor - John Doe\nActor - Jane Smith\nActor - Mike Johnson\n", result);
    }
}
