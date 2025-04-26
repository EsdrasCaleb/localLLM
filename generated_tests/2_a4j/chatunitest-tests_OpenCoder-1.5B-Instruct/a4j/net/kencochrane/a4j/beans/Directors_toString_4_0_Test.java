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

    @Test
    public void testToString() {
        // Create a mock Directors object
        Directors directors = mock(Directors.class);
        ArrayList<String> directorsList = new ArrayList<>();
        directorsList.add("John Doe");
        directorsList.add("Jane Smith");
        when(directors.getDirectorsArray()).thenReturn(directorsList);
        // Call the toString() method and store the result
        String expected = "# of Directors = 2\nDirector - John Doe\nDirector - Jane Smith";
        String actual = directors.toString();
        // Assert that the actual result matches the expected result
        assertEquals(expected, actual);
    }
}
