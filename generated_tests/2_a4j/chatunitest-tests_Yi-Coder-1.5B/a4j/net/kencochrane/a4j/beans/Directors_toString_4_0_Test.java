package net.kencochrane.a4j.beans;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;
import java.util.ArrayList;

@ExtendWith(MockitoExtension.class)
public class Directors_toString_4_0_Test {

    // Test class
    @Test
    public void toStringTest() {
        Directors directors = new Directors();
        directors.setDirector(new String[] { "John", "Jane", "Jill" });
        assertEquals("# of Directors = 3\nDirector - John\nDirector - Jane\nDirector - Jill\n", directors.toString());
    }
}
