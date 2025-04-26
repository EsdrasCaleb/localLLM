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

class Starring_getActor_3_0_Test {

    @Test
    public void testGetActor() throws NoSuchFieldException, IllegalAccessException {
        // Create an instance of Starring
        Starring starring = new Starring();
        // Add some actors to the ArrayList
        starring.setActor(new String[] { "Tom Hanks", "Meryl Streep", "Leonardo DiCaprio" });
        // Get the field 'actors' using reflection
        Field actorsField = Starring.class.getDeclaredField("actors");
        actorsField.setAccessible(true);
        // Get the ArrayList from the field
        ArrayList<String> actors = (ArrayList<String>) actorsField.get(starring);
        // Test the getActor method with different indices
        assertEquals("Tom Hanks", starring.getActor(0));
        assertNull(starring.getActor(-1));
        assertEquals("Meryl Streep", starring.getActor(1));
        assertNull(starring.getActor(2));
        // Index out of bounds
        assertNull(starring.getActor(3));
    }
}
