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

    private Starring starring;

    @BeforeEach
    public void setUp() {
        starring = new Starring();
    }

    @Test
    public void testToString_withActors() throws Exception {
        // Arrange
        String[] actorsArray = { "Actor1", "Actor2", "Actor3" };
        starring.setActor(actorsArray);
        // Act
        String result = starring.toString();
        // Assert
        String expected = "# of Actors = 3\n" + "Actor - Actor1\n" + "Actor - Actor2\n" + "Actor - Actor3\n";
        assertEquals(expected, result);
    }

    @Test
    public void testToString_withNullActors() throws Exception {
        // Arrange
        setPrivateField(starring, "actors", null);
        // Act
        String result = starring.toString();
        // Assert
        String expected = "Actors is null or size 0\n";
        assertEquals(expected, result);
    }

    @Test
    public void testToString_withEmptyActors() throws Exception {
        // Arrange
        setPrivateField(starring, "actors", new ArrayList<>());
        // Act
        String result = starring.toString();
        // Assert
        String expected = "Actors is null or size 0\n";
        assertEquals(expected, result);
    }

    private void setPrivateField(Starring starring, String fieldName, Object value) throws Exception {
        Field field = Starring.class.getDeclaredField(fieldName);
        field.setAccessible(true);
        field.set(starring, value);
    }
}
