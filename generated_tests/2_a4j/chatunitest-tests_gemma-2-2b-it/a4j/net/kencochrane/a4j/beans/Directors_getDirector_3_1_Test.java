package net.kencochrane.a4j.beans;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;
import java.util.ArrayList;

public class Directors_getDirector_3_1_Test {

    @Test
    void testGetDirector() {
        Directors directors = new Directors();
        directors.setDirector(new String[] { "John", "Jane", "Peter" });
        assertEquals("John", directors.getDirector(0));
        assertEquals("Jane", directors.getDirector(1));
        assertEquals("Peter", directors.getDirector(2));
    }
}
