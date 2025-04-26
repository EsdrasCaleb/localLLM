package net.kencochrane.a4j.beans;

import java.util.ArrayList;
import java.util.Arrays;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;

public class Directors_getDirector_3_0_Test {

    @Test
    public void testGetDirector() {
        Directors directors = new Directors();
        directors.setDirector(new String[] { "Director1", "Director2", "Director3" });
        assertEquals("Director1", directors.getDirector(0));
        assertEquals("Director2", directors.getDirector(1));
        assertEquals("Director3", directors.getDirector(2));
    }
}
