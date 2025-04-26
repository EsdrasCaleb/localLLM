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

public class Directors_toString_4_0_Test {

    @Test
    public void testToString() throws Exception {
        Directors directors = new Directors();
        // Test with empty director list
        String expected = "Director is null or size 0\n";
        Field field = Directors.class.getDeclaredField("directors");
        field.setAccessible(true);
        field.set(directors, null);
        assertEquals(expected, directors.toString());
        // Test with non-empty director list
        directors.setDirector(new String[] { "Director 1", "Director 2", "Director 3" });
        expected = "# of Directors = 3\nDirector - Director 1\nDirector - Director 2\nDirector - Director 3\n";
        assertEquals(expected, directors.toString());
    }
}
