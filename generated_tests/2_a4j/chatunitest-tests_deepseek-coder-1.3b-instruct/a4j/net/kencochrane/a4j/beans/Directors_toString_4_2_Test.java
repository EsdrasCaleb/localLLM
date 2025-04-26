package net.kencochrane.a4j.beans;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;
import java.util.ArrayList;

public class Directors_toString_4_2_Test {

    @Test
    public void testToString() {
        Directors directors = new Directors();
        String mockString = "Mock Director";
        directors.setDirector(new String[] { mockString });
        String expected = "# of Directors = 1\nDirector - Mock Director\n";
        assertEquals(expected, directors.toString());
    }
}
