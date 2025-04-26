package net.kencochrane.a4j.beans;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;
import java.util.ArrayList;

public class Authors_getAuthor_3_1_Test {

    @Test
    public void testGetAuthor_InValidIndex_ReturnsNull() {
        Authors authors = new Authors();
        authors.author = new ArrayList<>();
        String result = authors.getAuthor(-1);
        assertNull(result);
    }
}
