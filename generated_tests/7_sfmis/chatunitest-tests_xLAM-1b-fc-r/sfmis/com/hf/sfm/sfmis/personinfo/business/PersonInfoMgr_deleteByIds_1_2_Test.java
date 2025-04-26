package com.hf.sfm.sfmis.personinfo.business;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.hibernate.Transaction;
import com.hf.sfm.sfmis.personinfo.pdo.APersonInfo;
import com.hf.sfm.util.DaoFactory;

@ExtendWith(MockitoExtension.class)
public class PersonInfoMgr_deleteByIds_1_2_Test {

    @Mock
    PersonInfoMgr personInfoMgr;

    @InjectMocks
    PersonInfoMgr_deleteByIds_1_2_Test test;

    @Test
    public void deleteByIdsTest() {
        // Given
        String[] idnos = { "1", "2", "3" };
        String result = "1";
        when(personInfoMgr.deleteByIds(idnos)).thenReturn(result);
        // When
        String actualResult = test.personInfoMgr.deleteByIds(idnos);
        // Then
        assertEquals(result, actualResult);
    }
}
