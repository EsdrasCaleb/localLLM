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

    Directors directors = new Directors();

    @Test
    public void testGetDirector() {
        ArrayList<String> testList = new ArrayList<>(Arrays.asList("Director1", "Director2", "Director3"));
        directors.setDirector(testList.toArray(new String[0]));
        assertEquals("Director2", directors.getDirector(1));
        assertEquals("Director3", directors.getDirector(2));
        assertEquals(null, directors.getDirector(3));
    }
}
