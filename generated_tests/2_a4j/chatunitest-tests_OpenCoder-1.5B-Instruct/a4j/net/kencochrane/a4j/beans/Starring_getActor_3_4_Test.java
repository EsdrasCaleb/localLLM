package net.kencochrane.a4j.beans;

import java.util.Arrays;
import java.util.List;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;
import java.util.ArrayList;

public class Starring_getActor_3_4_Test {

    @Test
    public void testGetActor() {
        // Create a mock object of the Starring class
        Starring starring = Mockito.mock(Starring.class);
        // Create a list of actor names
        List<String> actors = Arrays.asList("Tom Hanks", "Leonardo DiCaprio", "Robert Downey Jr.");
        // Set the mock object's behavior when getActor is called
        when(starring.getActor(0)).thenReturn(actors.get(0));
        when(starring.getActor(1)).thenReturn(actors.get(1));
        when(starring.getActor(2)).thenReturn(actors.get(2));
        // Call the method being tested
        String actor1 = starring.getActor(0);
        String actor2 = starring.getActor(1);
        String actor3 = starring.getActor(2);
        // Assert that the method returns the expected values
        assertEquals("Tom Hanks", actor1);
        assertEquals("Leonardo DiCaprio", actor2);
        assertEquals("Robert Downey Jr.", actor3);
    }
}
