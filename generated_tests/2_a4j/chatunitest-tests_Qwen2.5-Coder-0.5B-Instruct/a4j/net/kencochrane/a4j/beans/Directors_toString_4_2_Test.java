package net.kencochrane.a4j.beans;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;
import java.util.ArrayList;

class Directors_toString_4_2_Test {

    @Test
    public void testToString() {
        Directors directors = new Directors();
        directors.setDirector(new String[] { "John Doe", "Jane Smith", "Alice Johnson" });
        assertEquals(directors.toString(), "Director - John Doe\nDirector - Jane Smith\nDirector - Alice Johnson");
    }
}
