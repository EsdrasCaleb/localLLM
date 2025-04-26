package net.kencochrane.a4j.beans;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;
import java.util.ArrayList;

public class Directors_toString_4_0_Test {

    @Test
    void testToString() {
        Directors directors = new Directors();
        directors.setDirector(new String[] { "John", "Jane" });
        String result = directors.toString();
        assertEquals("Director is null or size 0", result);
    }
}
