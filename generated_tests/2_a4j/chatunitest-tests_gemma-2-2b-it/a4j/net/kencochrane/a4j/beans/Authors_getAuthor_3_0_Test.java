package net.kencochrane.a4j.beans;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;
import java.util.ArrayList;

public class Authors_getAuthor_3_0_Test {

    @Test
    void testGetAuthor() {
        Authors authors = new Authors();
        authors.setAuthor(new String[] { "John Doe", "Jane Doe" });
        int index = 1;
        String expected = "Jane Doe";
        String actual = authors.getAuthor(index);
        assertEquals(expected, actual);
    }
}
