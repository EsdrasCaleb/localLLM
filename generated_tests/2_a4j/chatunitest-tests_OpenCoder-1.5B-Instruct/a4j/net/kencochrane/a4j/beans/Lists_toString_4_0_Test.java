package net.kencochrane.a4j.beans;

import org.junit.Test;
import static org.junit.Assert.assertEquals;
import java.util.Arrays;
import java.util.List;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;
import java.util.ArrayList;

public class Lists_toString_4_0_Test {

    @Test
    public void testToString() {
        Lists lists = new Lists();
        lists.setListId(new String[] { "List1", null, "List3" });
        String expected = "# of Lists = 3\n" + "list - List1\n" + "list - null\n" + "list - List3\n";
        String actual = lists.toString();
        assertEquals(expected, actual);
    }
}
