package com.hf.sfm.sfmis.personinfo.business;

// PersonInfoMgrTest.java
import org.junit.Test;
import org.junit.runner.RunWith;
import org.mockito.runners.MockitoJUnitRunner;
import java.util.Arrays;
import java.util.List;
import static org.junit.Assert.assertEquals;
import static org.mockito.Matchers.any;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import org.hibernate.Transaction;
import com.hf.sfm.sfmis.personinfo.pdo.APersonInfo;
import com.hf.sfm.util.DaoFactory;

@RunWith(MockitoJUnitRunner.class)
public class PersonInfoMgr_deleteByIds_1_4_Test {

    @Mock
    private PersonInfoMgr personInfoMgr;

    @Mock
    private List<String> list;

    @Test
    public void testDeleteByIds() {
        String[] idnos = { "1", "2", "3" };
        when(personInfoMgr.deleteByIds(idnos)).thenReturn("1");
        assertEquals("1", personInfoMgr.deleteByIds(idnos));
    }
}
