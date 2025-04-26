package net.kencochrane.a4j.beans;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;
import java.util.ArrayList;

public class Directors_toString_4_4_Test {

    @Test
    public void testToString() {
        // Mock the Directors ArrayList
        Directors mockDirectors = new Directors();
        mockDirectors.setDirector(new String[] { "Tom", "Dick", "Harry" });
        // Call the toString method
        String result = mockDirectors.toString();
        // Assert the result
        assertEquals("# of Directors = 3\nDirector - Tom\nDirector - Dick\nDirector - Harry\n", result);
    }
}
